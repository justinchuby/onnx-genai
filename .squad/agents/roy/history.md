# roy — History

## Role
Architecture/planning and implementation reviewer spanning engine phases, ORT2 shape/optimizer work, EPContext, packaging, and router design. Honor reviewer lockouts and keep documented contracts aligned with executable behavior.

## Summary through 2026-07-14T20:05:00Z

### Runtime roadmap
Planned and coordinated the initial engine phases: real ORT execution, paged/prefix KV, multi-session APIs, speculative decoding, tiered/quantized KV, pipeline/tool use, static-cache long context, and architecture decomposition. The engine should continue moving away from a monolith toward explicit backend/sampler/proposer seams.

### Router
Delivered the later §34 R2/R3 router core and reverse proxy, including affinity/load policies, persisted session mapping, SSE proxying, polling, metrics, drain/rebalance endpoints, weighted affinity correction, concurrent polling, response caps, and overload guards. This work is recorded with 73 tests.

### ORT2 foundation and shape inference
Built the initial IR/session foundation and authored the shape-inference crate. Chew and Holden rejected transpose and overflow defects; Roy is locked out of that original artifact and Deckard's fixes are canonical. Later wired shape inference into loader/session, removed const-fold-lite, and preserved bert_toy conformance.

### Optimizer and fusions
Activated opt-in session optimization with default-off byte invariance. Added decline-to-fuse guards for LayerNorm and MatMul+Add. Reviewed FusedGemm, FusedAttention, and GELU with adversarial and parity checks; all approved. Maintain strict guards and separate fused-vs-unfused drift from reference conformance tolerances.

### EPContext and ONNX encoding
Authored §55 design, corrected external-file default, and implemented the loader EPContext path. Encoder v1 was rejected for EP-specific literals in generic encoding; Roy is locked out and Deckard v2 is canonical. Preserve byte-exact opaque payloads and model-agnostic generic layers.

### Packaging and review protocol
Rejected the original crate-reservation runbook due to a publication cycle; Deckard was locked out and Leon's path-only dev-dependency fix passed re-review. Recent CUDA Phase-2a SDPA/GQA review was green, including layout, GQA mapping, causal indexing, numerics, safety, and H200 execution.

## 2026-07-15T00:00:00Z — Cross-agent session update

- Hardened CUDA DLPack commit validation to compare raw device identity; GPU review findings are incorporated in the final DLPack wave.

### 2026-07-16T00:00:00Z — Performance-and-design wave
Authored CUDA Gather, Shape, and Constant kernels; coverage reached 65.

### 2026-07-16T00:00:02Z — MatMulNBits GEMV wave
- Landed the direct-int4 VNNI M=1 GEMV (`2095325`, reviewed follow-up `2d7c974`), streaming packed nibbles with block-32 scales rather than materializing int8 weights.
- MatMulNBits improved 16.45→14.15 ms; decode reached about 50 tok/s at 24 threads and about 28 tok/s at 96 threads. NUMA-aware scheduling and projection fusion are the pending next levers.


### 2026-07-16T00:00:00Z — Allocation-free fused RMSNorm decode
- Landed `de62f76`: direct-output contiguous-f32 `SkipSimplifiedLayerNormalization` fast path, retaining the scalar/broadcast/strided/statistics fallback.
- RMSNorm fell 1.113→0.742 ms/step (-33.3%); five paired runs improved decode 44.20→46.45 tok/s (+9.1%) with matching tokens and 413 CPU EP tests.

## 2026-07-16T00:00:00Z — CUDA M2 packed-GQA artifact
- Implemented `ad73494` packed QKV splitting, device RoPE, and alias-aware O(1)-per-new-token KV append for 14/2/64 GQA.
- Sebastian rejected the test for host-seeded cache coverage and unsupported PTX (5/6); strict lockout applied, and Wallace's subsequent repair is canonical.

## 2026-07-16T14:20:00Z — CUDA M3 device-resident KV cache
- Merged `398c536`: persistent aliased K/V allocations have stable pointers, zero KV H2D/D2H transfers, O(1) mask updates, capacity/valid-length separation, and configurable default max length 4096.
- Sebastian cleared M3 and confirmed the CPU/CUDA mismatch beginning at token 10 is pre-existing M2 numerical drift.

## 2026-07-16T17:00:38+0000 — CUDA M5 int4 GEMV decode
- Merged `1de9584`: direct packed-int4 M=1 CUDA GEMV avoids f32 weight expansion and improved decode by approximately 68–96%.
- Wallace 🟢 verified H200 parity and the unchanged Qwen decode contract.

## 2026-07-16T18:11:48+0000 — CUDA sub-4-bit GEMV

- Merged `cef7073`: static M=1 CUDA `BlockQuantizedMatMul` now decodes MXFP4 and IQ4_NL native blocks; other IQ formats remain CPU-placed.
- Wallace 🟢 cleared H200 packing, exact decoded-weight, and 124-test coverage.

## 2026-07-16T19:27:57+0000 — CUDA IQ super-block GEMV wave

- Merged `8bf113e`: CUDA static M=1 BlockQuantizedMatMul GEMV now supports IQ4_XS, IQ2_XXS, IQ3_XXS, IQ2_XS, IQ2_S, and IQ3_S; together with MXFP4/IQ4_NL this is 8/10 IQ formats. Shared audited grids/sign tables moved into `onnx-runtime-quantization`; Leon and Wallace cleared the refactor and H200 GPU validation. IQ1_S/IQ1_M remain the final GPU formats.

## 2026-07-16T19:27:57+0000 — CUDA IQ1 GEMV completion

- Merged `06c4c06`: static M=1 CUDA `BlockQuantizedMatMul` GEMV now supports IQ1_S and IQ1_M through the shared 2,048-entry `IQ1S_GRID`, completing all 10 GPU IQ/MXFP4 formats.
- M>1 and unknown formats remain CPU-routed; Wallace 🟢 cleared H200 bit-exact parity, the shared-grid hash, and the full validation gates.


## 2026-07-16T23:06:37+0000 — Native CUDA serving safety gate

- `559c46f` added native CUDA Engine/server selection but was 🔴 rejected: CUDA-only sessions could not serve the real 144-node sub-4-bit model. Roy is locked out of this revision artifact; Deckard's `fa30410` safety gate is canonical.

## 2026-07-16T23:30:00+0000 — CUDA MatMul stale-test correction

- Merged `3d19b72`: the unsupported MatMul regression now asserts the current Int64 CUDA EP error instead of obsolete Phase 2a wording.
- Wallace 🟢 cleared the exact failure path and 129/129 CUDA tests.

## 2026-07-14T00:00:00Z — QMoE landed

- The `com.microsoft::QMoE` CPU kernel landed after Nabil’s review cycle and the checked-arithmetic/addressability hardening revisions. Blockwise Q4/Q8 is enabled; IQ1/IQ2/int2 and sparse mixer remain follow-ups.

## 2026-07-17T02:24:32Z — QMoE int1/int2 support

- Landed `cdb4ee5`: CPU `com.microsoft::QMoE` now accepts 1/2/4/8-bit expert weights; 3-bit remains rejected because packed values would cross byte boundaries.
- 2026-07-19: Added phased captured-run error classification and real no-replay tests for PR #30 (`bd17d07`).
- 2026-07-19T07:55:00Z: The phase-aware captured-run classification and no-replay regression work remains integrated after PR #32's rebase.

## 2026-07-19T07:42:20Z — CSA Phase B B4 landed

- Landed device ratio-4 index scoring and deterministic top-k selection in `77a44a4`. Chew approved; 17/17 H200 GPU parity tests, including adversarial tie-break fixtures, are bit-exact.

## 2026-07-19T07:42:20Z — CSA B5 fix landed

- Fixed B5-1 by splitting ratio-128 and ratio-4 device-attention flags, preserving host-oracle fallback for five-output ratio-4 nodes, tightening no-bias parity to `max_ulp == 0`, and adding the regression test. Chew approved `1ddf01b`; 19/19 H200 tests passed.

## 2026-07-19T07:42:20Z — CSA B5 fix landed

- Fixed B5-1 by splitting ratio-128 and ratio-4 device-attention flags, preserving host-oracle fallback for five-output ratio-4 nodes, tightening no-bias parity to `max_ulp == 0`, and adding the regression test. Chew approved `1ddf01b`; 19/19 H200 tests passed.

## 2026-07-19T07:42:20Z — CSA Phase B B6 landed

- Landed `2a7703a`: CUDA-graph capture compatibility for ratio-4 fp8 6-output CSA, including device index replication, device-resident cursors, stable pooled workspaces, and a dedicated non-blocking compute stream.
- `cuda_graph_compatible()` is true only for that configuration; 20/20 CSA tests and the full ep-cuda suite passed on H200. Chew approved with non-blocking nits deferred to B7.

## 2026-07-19T07:42:20Z — CSA Phase B B7 landed

- Landed `d81b96a`, completing CSA Phase B B0–B7 with stream-ordered checkpoint/restore, device-default ratio-4 fp8 capture path, host fallback flag, and instance-scoped metrics. Chew approved with non-blocking N1/N2 follow-ups; 24 CSA tests plus 1 ignored MTP smoke passed and the full ep-cuda suite was green on H200.


## 2026-07-20T07:15Z — Decode bottleneck investigation

- Profiled native int4 decode: MatMulNBits was 83% of time and GQA full-KV materialization 14.7%; the M=1 scheduling finding informed sapper-48.


## 2026-07-20T13:35:00Z — Multistream performance and issue #40

- Landed guarded row-parallel CPU GQA (`c391327`): +8.6% 512-token decode, 13.9% lower profiled GQA time, and much faster prefill with bit-identical output; Luv approved.

## 2026-07-21T03:15:00Z — CUDA graph M4 validated
- Made all four normalization decode variants capture-safe with persistent shape-keyed SkipSimplified metadata; Chew approved and `6184d82` landed.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.

## 2026-07-21T05:40:00Z — fp16 decode and cross-platform reconciliation

- Completed OS-aware CUPTI discovery across Linux/macOS/Windows, including pip layouts and Windows ARM64 graceful degradation (`8cd36c3`); Pris approved.

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

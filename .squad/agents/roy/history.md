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

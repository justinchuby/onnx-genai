# batty — History

## Project context
Engine/EP implementer for the Rust ONNX runtime. Canonical ownership: ORT owns forward execution and physical KV; engine owns generation policy and logical KV. CPU kernels rely on session-side `strided::view_in_bounds` before dispatch.

## Summary through 2026-07-14T20:05:00Z

### Engine and KV foundations
Delivered generation, paged/prefix KV, constrained decoding, extensibility seams, prompt-lookup speculative decoding, SWA/sink hardening, mixed-layer KV groundwork, and early vision expansion. Metal-prefill hybrid was measured slower than CPU and should not be productionized. Connector KV import should eventually degrade to `Ok(None)` on import-runner failure.

### ORT2 EP and C API
Implemented the pure-Rust CPU EP foundation and expanded it through the Phase-1 kernel set; contributed the Phase-1 C ABI with opaque handles, panic fences, atomic run commit, and explicit error mapping. Hardening closed legacy Softmax semantics, NaN propagation, saturating casts, checked allocation geometry, dynamic-output guards, and shared Slice planning. Deckard completed the storage-byte overflow correction after Holden rejected the initial artifact.

### Optimizer and fused kernels
Generalized dispatch to `(domain, op_type)` and moved optimizer fusions to `com.microsoft`. Implemented executable, parity-tested LayerNorm, FusedMatMulBias, FusedGemm, FusedAttention, and Gelu paths with strict decline-to-fuse guards. bert_toy remained within reference tolerance; fusion Phase 2 is complete.

### EPContext
Implemented session consume and writer v1. Consume supports primary/reference resolution, external blobs, payload dedup, and executor bypass. Writer v1 was rejected for non-injective sidecar naming; Batty is locked out of that artifact. Leon and Gaff produced later revisions, with Gaff v3 merged as `0fa025e`. Remaining consume advisories include covered-node dedup, duplicate-primary diagnostics, and stronger traversal tests.

### Load-time validation
Unified validation behind `validate_model()` across disk/bytes and session load paths. Models now fail fast with actionable `UnsupportedControlFlow` or `DanglingTensorRef` errors. Empty graphs remain valid; per-kernel/shape-dependent checks remain dispatch concerns; IR invariants already enforced by `Graph::validate` are not duplicated. Holden reviewed 🟢; merged to `origin/main` as `2a99eec`.

### Reviewer lockouts and follow-ups
Batty is locked out of revising: H-D1 storage sizing, fusion follow-ups identified on earlier optimizer reviews, EPContext writer after v1 rejection, and other artifacts explicitly reassigned by reviewers. Preserve reviewer-protocol ownership when addressing advisories.

### 2026-07-16T00:00:00Z — Performance-and-design wave
Unified text codecs around TextCodec and renamed the text APIs.

### 2026-07-16T00:00:00Z — onnx-rs Python serialization bindings
Added the independent abi3-py310 `onnx-rs-python` crate, importing as `onnx_rs`, with opaque models, binary load/save, and text/JSON/TextProto codec functions (`1ae9a3d`). Freysa rejected the path conversion seam; Deckard's cleared path fix landed as `5b348b5`.


## 2026-07-16T19:27:57+0000Z — Native backend selector revision

Under Deckard's strict reviewer lockout, revised native serving in `2ae464b`: exact `com.github.onnxruntime.genai::BlockQuantizedMatMul` opset-v1 Auto detection, explicit errors for unsupported request speculation/pipelines/non-CPU selection, and regressions. Holden re-reviewed 🟢 CLEAR.

- 2026-07-18: Restored the pre-Phase-1 public MtpConfig struct contract via internal ResolvedMtpConfig; MTP Phase 1 re-review approved.
- 2026-07-19: Made BQMoE claim validation zero-allocation (`67abdb5`); hardened PR #30 retry safety and PR #34 capture gating.
- 2026-07-19T07:55:00Z: IndexShare v1's frozen ABI, exact CPU oracle, and interior-sentinel regression merged at `744a9a7`.


## 2026-07-19T07:42:20Z — CSA B2 nit fix landing

- Fixed B2 RMSNorm rounding parity and removed the redundant carry-reset loop in `2067504`; Chew re-reviewed 🟢 APPROVE and 14/14 GPU parity tests remained bit-exact.

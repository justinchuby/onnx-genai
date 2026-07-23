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

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.


## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.
- 2026-07-21T23:55Z — Native CUDA opset-24 ConstantOfShape/Gelu/OneHot landed; WP4 revision is the active correction after Zhora/Gaff lockout; clippy hygiene folded.

- 2026-07-22T00:00:00Z — CUDA graph auto-enable in native decode merged to main as `610bde0`; H200 Qwen2.5-0.5B improved 441.49→828.54 tok/s and Phi-4-mini 67.32→94.91 tok/s, token-exact with zero fallbacks. Leon reviewed 🟢.

## 2026-07-23T14:55:00Z — Mobius PR #404 QMoE emitter

- Extended `glm5.2-moe-export` with int4 QMoE emitter/packer at `751645b`: expert-major FC1/FC2 packing, swiglu gate/up interleave, asymmetric zp inputs 11/12, and GGUF requant-on-zp-mismatch. Chew approved with a Ruff syntax caveat under fix.

- 2026-07-23T15:45:00Z: Corrected DeepSeek low-precision routing in Mobius PR #404 (`bd88fa8`): `_router_logits()` widens both hidden states and gate weights to f32. Strict f16 CUDA smoke had zero fallbacks, 32 finite tokens, and 2.534 ms/token; f16/bf16 graph contracts cover mask, RoPE, and router dtypes.

## 2026-07-23T20:30:00Z — Real DeepSeek-V2-Lite artifact
- Exported a structurally valid full-depth real f16/int4 QMoE artifact: 27 layers, 26 QMoE nodes, 189 MatMulNBits nodes, asymmetric block-128 quantization, and MLA K/V widths 192/128.
- Native semantic execution awaits the in-flight block-32 re-export or native block-128 dense MatMulNBits support.

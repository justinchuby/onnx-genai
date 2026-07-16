# fact-checker — History

## Project Context (day 1)
- **Project:** onnx-genai — Rust inference runtime for generative AI on ONNX Runtime.
- **Stack:** Rust edition 2024, Cargo workspace, ORT backend, HF tokenizers.
- **Crates:** onnx-genai, -metadata, -kv, -scheduler, -engine, -ort, -server.
- **Requested by:** Justin Chu
- **Team formed:** 2026-07-12


## 2026-07-14T06:06:00Z — EPContext §55/§21.4 Verification

- Verified Roy's EPContext node design (`squad/ort2-epcontext-design` @ c48f5c4) against authoritative ORT source (contrib_defs.cc, session_options_config_keys.h, ep_context_options.cc, QNN/OpenVINO EP source).
- **Result: 🟡 SHIP-with-one-required-fix.** All 10 op attributes exact; session-option key strings exact; embed_mode/main_context semantics correct; model-agnostic dispatch verified.
- **❌ Required fix found:** §21.4 `ep.context_embed_mode` default stated as `1`; ORT runtime default is `0` (ep_context_options.cc:40). Roy applied fix in roy-11 (merged cf614e4).
- Advisory: TOC/header numbering mismatch (pre-existing, not introduced by this change).

## 2026-07-15T01:52:00Z — Session update

- Fact-checked KV insertion: ORT GQA shared-buffer is sanctioned, standard ONNX Attention now has cache semantics, and HF calls cache.update() inside attention.

### 2026-07-16T00:00:00Z — Performance-and-design wave
Verified model-package external claims; corrections were applied.

### 2026-07-16T00:00:03Z — Projection-fusion fact check
Verified the model-specific claims in `docs/PROJECTION_FUSION.md`: QKV is already packed, while 24 gate/up pairs are separate `4864|4864→9728` candidates. Confirmed the executor seam and packing math; documented that 124.6875 MiB of fused B+scale payload does not bound actual RSS because alignment copies may add memory.

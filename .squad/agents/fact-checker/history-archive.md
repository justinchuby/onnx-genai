# Fact Checker — History Archive

## Archived 2026-07-29 (full pre-compaction snapshot)

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
Verified the model-specific claims in `docs/quantization/PROJECTION_FUSION.md`: QKV is already packed, while 24 gate/up pairs are separate `4864|4864→9728` candidates. Confirmed the executor seam and packing math; documented that 124.6875 MiB of fused B+scale payload does not bound actual RSS because alignment copies may add memory.

### 2026-07-16T00:00:00Z — Native CUDA decode design fact check
Audited `docs/execution/NATIVE_CUDA_DECODE.md`: 14 central claims verified, including concrete CPU EP wiring, object-safe dynamic EP dispatch, packed-QKV GQA and O(capacity) KV blockers, and cudarc graph APIs. Required M4 corrections—a real non-null stream and serialized ownership of non-Send/Sync CUDA graphs—were incorporated in `33beb8d`; virtual-dispatch cost remains unmeasured.

## 2026-07-27T09:15:14-07:00 — CLI competitive/devil's advocate research

- Verified `onnx-genai` CLI surface against `crates\onnx-genai-cli\src\lib.rs`, `commands.rs`, and server `ServeArgs`.
- Compared current CLI UX against Ollama, llama.cpp, vLLM, mlx_lm, LM Studio `lms`, and Microsoft `onnxruntime-genai` builder/API docs.
- Findings written to `docs\research\cli\03-competitive-and-devils-advocate.md`: top gaps are model acquisition, conversion/quant/fine-tuning, and benchmark/batch commands; strongest counter-argument is that CLI polish may distract from CUDA/perf/model enablement.
### 2026-07-27T09:17:00Z — Win verification: "Native CPU EP beats ORT by 1.27×"
- **Claim:** Iran reports native FP16 at 57.5 tok/s = 1.27× ORT's best (45.0 FP32). PR #227 headline.
- **Result: ❌ OVERSTATED — cannot reproduce.** Independent reproduction with same harness (`compare.rs`), model, prompt, flags yields native FP16 median 36.1 tok/s (best single: 42.7). ORT FP32 matches at 45.7 tok/s. Native/ORT ratio is 0.79× on decode, 0.31× end-to-end.
- Three of four cells reproduce (ORT FP32, ORT FP16, Native FP32). The native FP16 headline cell does not.
- FP16 GEMV path confirmed active (atomic probe). Output coherence confirmed at 100 tokens (identical to FP32). Non-determinism found at 500 tokens (SPMD auto-calibration).
- 906 tests pass; fmt clean.
- Filed `.squad/decisions/inbox/fact-checker-win-verification.md`.

### 2026-07-27T10:55:00Z — Re-verification: calibrator hypothesis confirmed
- **Iran's explanation:** SPMD auto-calibrator mis-sampled under overnight machine load, committing to the flat Rayon path. This specifically devastates native FP16, which depends on the SPMD pool for multi-threaded streaming.
- **Result: ✅ TRUE-WITH-CAVEATS.** Re-ran all four cells on a quiet machine:
  - Native FP16: **58.69 tok/s** [58.15, 60.54] — reproduces Iran's ~59.78
  - ORT FP32: **45.76 tok/s** [45.67, 46.07]
  - Ratio: **1.28× ORT's best** (decode-only)
- **Calibrator hypothesis confirmed by 3-way experiment:** forced-pool=60.20, auto(quiet)=58.69, forced-flat=43.78, auto(loaded)=24.56 tok/s
- **Non-determinism root-caused:** auto-calibrator re-probing mid-generation causes path switching → FP non-associativity → token divergence at ~459 tokens. Forced-pool and forced-flat are individually deterministic. At ≤200 tokens, pool and flat produce identical tokens.
- **Caveats:** decode-only (end-to-end 0.42× ORT); TTFT 10× worse; requires quiet host or PERSISTENT_POOL=1; Qwen 0.5B only.
- Updated `.squad/decisions/inbox/fact-checker-win-verification.md` with re-verification section.

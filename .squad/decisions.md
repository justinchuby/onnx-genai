# Decisions

> Current decision ledger. The prior reconciliation ledger is preserved in
> `.squad/decisions-archive/2026-07-23T20-30-00Z-pre-parity-and-deepseek-ledger.md`.

## Index

- `2026-07-23T00-00-00Z-pre-reconciliation-ledger.md`: earlier processed inbox source records.
- `2026-07.md`: monthly historical ledger.
- `2026-07-23T20-30-00Z-pre-parity-and-deepseek-ledger.md`: archived pre-parity/DeepSeek active ledger.

## 2026-07-23 — Native/ORT parity and real DeepSeek-V2-Lite

### Lock scoped native/ORT decode parity regression coverage
**By:** Roy; reviewed by Fact Checker and Gaff
**What:** `scripts/check_native_ort_parity.py` and `tests/parity/` lock 128-token native/ORT CUDA parity for Phi-4-mini and Qwen2.5-0.5B. Qwen2.5-1.5B and 7B lock their observed first divergences and require native's token to match an independent f32 oracle dequantizing the exact deployed symmetric block-32 Q4 weights.
**Why:** The evidence establishes exact parity for the two recorded fixtures and native agreement with the deployed-Q4 oracle at the two measured divergence positions. It does not claim global numerical equivalence or blanket backend superiority.

### Keep parity-oracle scope explicit and harden before package changes
**By:** Gaff and Fact Checker
**What:** The current Q4 oracle is correct for the locked Qwen packages: low-nibble-first packing, implicit zero point 8, float16 scales, block size 32, no explicit zero points, and no `g_idx`. Future package expansion must add graph-contract guards or generalize dequantization; keep oracle provenance independently captured and assert its split relationship explicitly.
**Why:** ORT-CPU agreement alone is not an independent ground truth. The exact-Q4 f32 oracle supports the bounded observed claims, while preventing unsupported extrapolation to different artifacts or later autoregressive positions.

### Record the real DeepSeek-V2-Lite int4 export contract
**By:** Batty
**What:** A real bf16 checkpoint was exported to f16/int4 ONNX with 27 decoder layers, 26 QMoE nodes, 27 Attention nodes, and 189 MatMulNBits nodes. It uses asymmetric block-128 quantization, 64 routed experts with top-6 routing, and MLA widths of K=192/V=128; ONNX structural validation passed.
**Why:** This is a full-depth real-weight artifact suitable for native semantic validation, superseding synthetic-only structural evidence.

### Block real DeepSeek native semantic conclusions on block-128 support
**By:** Marsten
**What:** The real artifact fails before token 1 at layer-0 `q_proj`: strict CUDA placement has zero CPU fallbacks, but native fp16 `MatMulNBits` accepts only the block-32 packed layout while all 189 dense nodes are block-128. QMoE and MLA are not reached, so no semantic conclusion about them is valid.
**Why:** Resolve the layout incompatibility by re-exporting dense projections as block-32 or adding native fp16 block-128 MatMulNBits support. The latter is in flight; a block-32 re-export is also in flight.

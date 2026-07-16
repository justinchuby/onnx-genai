# Joi — History

## 2026-07-16T19:05:18+0000 — CPU BlockQuantizedMatMul prefill optimization

- Merged `5010261`: K-panel-parallel dequantization for all formats, bit-exact AVX2 paths for MXFP4/IQ4_NL/IQ4_XS, and adaptive generic GEMM row scheduling.
- At M=64/K=4096/N=4096, generic matmul improved 32–35×; all ten formats remain scalar-bit-exact. Leon 🟢 cleared default and oneDNN CPU EP validation.

## 2026-07-16T23:30:00+0000 — Pad opset-18 axes inference fix

- Merged `0a105a4`: Pad inference applies begin/end values to the optional axes subset, including negative axes, yielding `[2,3,4,6]` / 576 bytes for expanded Attention.
- Bryant 🟢 cleared the regression and suites; execution now exposes the separate `Less` Bool-dtype inference follow-up.

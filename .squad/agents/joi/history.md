# Joi — History

## 2026-07-16T19:05:18+0000 — CPU BlockQuantizedMatMul prefill optimization

- Merged `5010261`: K-panel-parallel dequantization for all formats, bit-exact AVX2 paths for MXFP4/IQ4_NL/IQ4_XS, and adaptive generic GEMM row scheduling.
- At M=64/K=4096/N=4096, generic matmul improved 32–35×; all ten formats remain scalar-bit-exact. Leon 🟢 cleared default and oneDNN CPU EP validation.

## 2026-07-16T23:30:00+0000 — Pad opset-18 axes inference fix

- Merged `0a105a4`: Pad inference applies begin/end values to the optional axes subset, including negative axes, yielding `[2,3,4,6]` / 576 bytes for expanded Attention.
- Bryant 🟢 cleared the regression and suites; execution now exposes the separate `Less` Bool-dtype inference follow-up.

## 2026-07-17T00:19:41+0000 — CPU ONNX Mod

- Merged `aa7127e`: CPU `Mod` supports fmod modes, broadcasting, and NumPy floor-mod integer semantics; 13/13 official Mod CPU cases passed.
- Expanded Attention now reaches missing `And` execution at node 39; direct BF16 Mod coverage remains a follow-up.

## 2026-07-17T02:24:32Z — Standard shape-inference expansion

- Landed `98ee7a6`: added OneHot, Trilu, DepthToSpace, SpaceToDepth, and Compress inference with safe symbolic and checked-arithmetic behavior; 140 tests passed.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.

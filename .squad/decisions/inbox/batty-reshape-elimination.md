# Batty — load-time reshape/view dispatch elimination

Date: 2026-07-29T18:10:00-07:00
Branch: qwen3-perf-followups

Implemented two layers for native decode shape metadata overhead:

- `onnx-runtime-optimizer::ShapeNoOpElimination` rewires provably identical `Identity` / `Reshape` / `Squeeze` / `Unsqueeze` nodes during EP optimization, while preserving graph outputs, initializer-backed malformed values, unknown shapes, and control-flow captures.
- The session executor now marks `Reshape` / `Squeeze` / `Unsqueeze` as executor-owned static view nodes in the load-time plan. They still resolve data-dependent output shapes at their topological point, but never call EP `get_kernel` or dispatch a kernel/copy.

Validation:

- `cargo test -p onnx-runtime-optimizer shape_noop --quiet` passed.
- `cargo test -p onnx-runtime-ep-cpu optimizer --quiet` passed.
- `cargo test -p onnx-runtime-session executor --quiet` passed.
- `cargo test -p onnx-genai-engine --features native-backend,mlas native_decode --quiet` passed.
- Real qwen3 divergence lock with `QWEN3_0_6B_E2E_DIR=C:\Users\justinchu\.foundry\cache\models\Microsoft\qwen3-0.6b-generic-cpu-4\v4` did not reach decode: existing metadata ambiguity (`model.io.token_input` matches both `input_ids` and `attention_mask`).

Profile result on qwen3-0.6b native CPU (`ONNX_GENAI_PROFILE_OPS=1`, `--steady --runs 1 --tokens 16`): `Reshape` disappeared from the per-op table (previous coordinator baseline: 113 calls / ~0.86 ms; intermediate strict no-op pass only: 56 calls / ~0.16 ms). Representative final decode step: MatMulNBits 7.223 ms (197 calls), GQA 0.564 ms, SkipSimplifiedLayerNormalization 0.212 ms, SimplifiedLayerNormalization 0.195 ms, FusedSiluMul 0.144 ms; no Reshape row.

Benchmark (`profile_native --backend {native,ort} --ep cpu --steady --runs 12 --tokens 96`, qwen3-0.6b, contended Windows host):

- Native best 106.30 tok/s, median 99.28 tok/s (repeat run; first run best 100.58, median 96.12).
- ORT best 108.18 tok/s, median 106.97 tok/s.

Honest gate: Reshape dispatch is gone, but native did not beat ORT in this window. Best-case residual is ~1.9 tok/s vs ORT best; median is still worse under host contention.

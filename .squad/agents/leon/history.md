# Leon — History

## Role and invariants

Engine/KV/runtime-buffer implementer. Runtime owns KV; model geometry is sourced from
`inference_metadata.yaml`, not ORT-GenAI configuration. Preserve device-buffer ownership,
past/present aliasing, exact real-model comparison settings, and reviewer lockouts.

## Historical summary through 2026-07-28

- Generalized shared KV, attention-sink SWA, connectors, and prefix payload
  materialization; equal prefix keys now prove equal content. Remaining work is
  multi-layer fixtures, graceful recomputation, and heterogeneous connector payloads.
- Delivered heterogeneous Gemma4 E2B speculative execution and corrected proposer inputs
  to `embed(last_token) + last_hidden`; correctness improved while performance tuning remains
  separate.
- Hardened loaders and fusion: unsupported dtypes fail closed; LayerNorm operand ordering is
  guarded; opset imports validate recursively; the `nxrt_*` C ABI replaced `ort2_*`.
- Implemented weight-offload foundations, route-first QMoE selection, CUDA
  `SparseKvGather` D==0 validation, and CPU CSA claim validation.
- Contributed to CUDA graph/capture correctness (SequenceAt/Scan parity, Phi decode lock,
  default-domain Attention and standalone RoPE capture regressions) and to the #291 rewind
  policy split: public runner rewind rejects before mutation while internal speculative rewind
  remains permitted.
- Unified native CUDA and ORT KV capacity policy in `onnx-genai-kv`, including transactional
  growth and injected allocation/copy/mask/capture-failure tests. Native CUDA growth is a
  graph boundary: preserve the prefix, invalidate stale capture, and recapture after growth.
  Real DeepSeek CUDA validation verified 4→8→16 growth and recapture; no speculative
  free-memory ceiling is imposed.

## 2026-07-29T03:45:00+0000 — PR #382 CPU shared-buffer regression lock

- Under Benny's reviewer lockout, added
  `cpu_shared_buffer_continuous_batch_uses_declared_kv_pairs`, using
  `tiny-llm-sharedbuffer` and explicit float32 KV metadata.
- The engine-level CPU test runs continuous batching and compares sequential generation; it
  fails at session construction if declared `model.io.kv_inputs` / `kv_outputs` are not
  threaded to `BatchedSharedBufferDecodeSession`.
- Revert verification proved the test catches the latent #380 regression previously hidden
  because the equivalent CUDA E2E auto-skips without CUDA. The repair and test merged in
  `85b9ba15`.

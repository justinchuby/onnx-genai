# Leon — History

## Role and invariants

Engine/KV/runtime-buffer implementer. Runtime owns KV; model geometry is sourced from
`inference_metadata.yaml`, not ORT-GenAI configuration. Preserve device-buffer ownership,
past/present aliasing, exact real-model comparison settings, and reviewer lockouts.

## Historical summary through 2026-07-29

- Generalized shared KV, attention-sink SWA, connectors, prefix payload materialization, and
  heterogeneous Gemma4 E2B speculative execution. Remaining work includes multi-layer
  fixtures, graceful recomputation, and heterogeneous connector payloads.
- Hardened loaders and fusion: unsupported dtypes fail closed; LayerNorm operand ordering and
  recursive opset imports are guarded; the `nxrt_*` C ABI replaced `ort2_*`.
- Delivered route-first QMoE, CUDA `SparseKvGather` D==0 validation, CPU CSA claim validation,
  and the #291 rewind policy split. Native CUDA and ORT now share transactional KV-capacity
  growth; CUDA growth invalidates and later recaptures graphs while preserving the prefix.
- PR #382 added CPU continuous-batch shared-buffer coverage with declared KV pairs, catching
  the construction failure hidden by CUDA E2E auto-skips.
- The DeepSeek CUDA/ORT repetition investigation was corrected: both backends reproduce the
  same greedy-decoding degeneration. The native 4096-capacity error is downstream, not its
  cause. Model-declared sampling defaults therefore follow strict precedence: explicit caller
  setting, then model declaration, then greedy fallback.
- CUDA driver API availability comes from the display driver; CUDA 12 fallback is supported by
  both the EP loader and cudarc. Verify inferred environment claims directly before recording
  a blocker.

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

## 2026-07-28T18:00:00-0700 — PR #385 re-scoped onto #392 (server + Python sampling wiring)

- #392 merged the engine + CLI half of the model-sampling-defaults work to `main`
  (`resolve_sampling_defaults`, `Option`-typed `SamplingOverrides`, CLI wiring). Confirmed #392
  preserved the strict precedence (explicit override > model-declared > greedy fallback) and the
  three-state `Option` typing — no design regression to raise.
- Reset the branch onto `origin/main` and re-applied ONLY the delta #392 left missing (the two
  Copilot findings): server + Python wiring, the misnamed-test fix, and a resolver-level
  temperature-0 → greedy guard. Dropped everything already on `main`. Final diff: 7 files,
  +414/-49, single commit `b78d8bec`.
- Resolution stays at each front end's request-construction boundary (CLI already via #392;
  server via `ModelHandle::generation_defaults` in `prepare_generate_request`/`prepare_completion`;
  Python via `engine.metadata().generation`). Not the engine, because `GenerateOptions` erases
  explicit-vs-unspecified (RULES rule 5). Pipelines + audio pass `None` (no-op).
- Finding 2: renamed `explicit_temperature_zero_forces_greedy` →
  `explicit_greedy_override_is_applied_and_keeps_its_temperature`; the resolver now owns the
  `temperature == 0` → greedy mapping for every consumer (new test
  `resolved_temperature_zero_forces_greedy_without_explicit_greedy`). `temperature: Some(0.0)`
  without greedy = deterministic argmax; sampler never zero-divides (`TemperatureProcessor` only
  inserted when `temperature > 0.0 && != 1.0`).
- Behaviour change: server/Python callers that don't override sampling now decode stochastically
  against `do_sample: true` models, matching the CLI. No greedy-assuming test broke.
- Gates green: fmt --all clean; clippy -D warnings on engine/server/python/cli; engine lib 274,
  server sampling tests + 116 pass (1 pre-existing `vision.onnx`-fixture failure, identical on
  clean main), python 5, cli lib 103.
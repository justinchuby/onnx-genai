# Leon — History (compacted 2026-07-29)

**Role:** Engine/KV/runtime-buffer implementer. Runtime owns KV; model geometry from `inference_metadata.yaml`. Preserve device-buffer ownership, past/present aliasing, exact real-model comparison, reviewer lockouts.

**Historical summary through 2026-07-28:** Generalized shared KV, attention-sink SWA, connectors, prefix payload materialization (equal prefix keys prove content equality). Delivered heterogeneous Gemma4 E2B speculative execution (proposer inputs corrected). Hardened loaders/fusion (unsupported dtypes fail-closed, LayerNorm operand-order guarded, opset validation recursive, `nxrt_*` C ABI replaces `ort2_*`). Implemented weight-offload foundations, route-first QMoE, CUDA SparseKvGather D==0 validation, CPU CSA claim validation. Contributed to CUDA graph/capture correctness (SequenceAt/Scan parity, Phi decode lock, default-domain Attention and RoPE capture regressions) and PR #291 rewind policy split (public rewind rejects before mutation, internal speculative rewind allowed). Unified native CUDA/ORT KV capacity policy with transactional growth; real DeepSeek validation verified 4→8→16 growth/recapture.

Older detailed work (2026-07-14 through 2026-07-28) archived in `history-archive.md`.

## Recent work (2026-07-29)

### 2026-07-29T03:45:00+0000 — PR #382 CPU shared-buffer regression lock

- Under Benny's reviewer lockout, added `cpu_shared_buffer_continuous_batch_uses_declared_kv_pairs`, using `tiny-llm-sharedbuffer` and explicit float32 KV metadata.
- Engine-level CPU test runs continuous batching and compares sequential generation; fails at session construction if declared `model.io.kv_inputs` / `kv_outputs` stop reaching `BatchedSharedBufferDecodeSession`.
- Revert verification proved the test catches latent #380 regression previously hidden because equivalent CUDA E2E auto-skips without CUDA. Repair and test merged in `85b9ba15`.

### 2026-07-28T18:00:00-0700 — PR #385 re-scoped onto #392 (server + Python sampling wiring)

- #392 merged engine + CLI half of model-sampling-defaults work to `main` (`resolve_sampling_defaults`, `Option`-typed `SamplingOverrides`, CLI wiring). Strict precedence preserved: explicit override > model-declared > greedy fallback.
- Reset branch onto `origin/main`, re-applied only the delta #392 left missing: server + Python wiring, misnamed-test fix, resolver-level temperature-0 → greedy guard. Final diff: 7 files, +414/-49, single commit `b78d8bec`.
- Server/Python callers now decode stochastically against `do_sample: true` models, matching CLI. No greedy-assuming test broke.
- Gates green: engine lib 274, server sampling 116 pass (1 pre-existing fixture failure)
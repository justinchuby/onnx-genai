# Leon — History (compacted 2026-08-27T17:00:00Z)

**Role:** Engine/KV/runtime-buffer implementer. Runtime owns KV; model geometry comes from `inference_metadata.yaml`. Preserve device-buffer ownership, past/present aliasing, exact real-model comparison, and reviewer lockouts.

## Durable lessons
- Device transfer and KV growth contracts must be enforced at the production binding boundary, not only at admission or prefill.
- A flag set on only some paths defaults conservatively; the #31974 `is_packed` flip broke float LayerNorm.
- CUDA pointer/index products promote before their first multiplication; checked host byte geometry does not make completed i32 products safe.
- Dynamic output selection has one policy authority and one materialization; capture declines when cardinality requires synchronous host resolution.
- Synchronous pageable H2D completes device residency before return and orders against the nonblocking compute stream; asynchronous overlap stays in `htod_async`.
- Teardown and FFI-sensitive paths remain non-panicking.

## Historical context

Shared/paged KV, capture, plugin compute hardening, fixture conversion, V2-Lite planner/capacity repair, and upstream LayerNorm work are archived. Full older history is in `history-archive.md`; the exact hot file before this compaction is in `history-archive-2026-08-27T17-00-00Z.md`.

## 2026-08-24 — CUDA DFT via cuFFT (PR #2080)

- Added f32 real/complex CUDA DFT: forward/inverse, full/onesided, arbitrary axis, and truncating/zero-padding dft_length.
- Added wheel-aware dynamic cuFFT loading, RAII stream-bound plans, a 16-entry LRU, and EP-governed packed/work workspace; capture fails closed.
- RTX 4060 CUDA 13.1 matrix: 7 passed, 0 failed, 0 ignored. Sign, inverse-scale, and N/2+1 mutations each failed their dedicated test.
- Verified `nvidia-cufft==12.1.0.78` + `nvidia-nvjitlink==13.1.115`; binaries remain external NVIDIA-licensed dependencies.
- Broader conformance coverage guard has four pre-existing missing profile entries (PagedAttention, Mish, Celu, TensorScatter); DFT is correctly profiled.

## 2026-08-25 — Batched CUDA STFT (PR #2091)

- Added f32 CUDA STFT v17 using #2080's shared cuFFT RAII plan/cache/workspace and #2083's CPU/shape semantics.
- One fused kernel extracts/windows all complete frames, one PlanMany call transforms batch×frames, and one kernel unpacks output; capture and strided inputs fail closed.
- RTX 4060 CUDA 13.1 suite: 9 passed, 0 failed, 0 ignored; DFT regression 7/7. Workload [2,16,1], N=5, step=3 used 4 frames/signal, FFT batch 8, 2 explicit NVRTC launches + 1 cuFFT call, 512 B governed workspace.
- Window-ignore, non-overlap-step, dropped-final-frame, and missing-Nyquist mutations each failed their dedicated tests.
- No throughput claim; Linux/H100/H200 and cuFFT internal launch count remain unmeasured.

## 2026-08-25 — CPU/plugin NonMaxSuppression KernelSizedOutput (PR #2112)

- Migrated NMS to #2101 host-only KernelSizedOutput; selection runs exactly once and returns owned Int64 `[selected,3]` bytes shared by native/plugin paths.
- Plugin now claims NMS via exact census; ORT E2E assigned NMS to cpu_ep and produced exact dynamic `[4,3]` rows with one 96-byte materialization copy.
- CPU targeted NMS: 8 passed; plugin census: 5; generic kernel-sized: 4; real NMS E2E: 1. Clippy passed.
- IoU, score-threshold, center-box, double-selection, skipped-copy, removed-census, and empty-census mutations were all caught.
- CUDA NMS remains deferred until the device-sized-output policy is established by CUDA Unique work.

## 2026-08-25 — Bounded CUDA NonMaxSuppression (PR #2130)

- Reused #2113 DeviceWorkspace and #2112 CPU NMS semantics; f32 static contiguous boxes/scores, <=256 boxes and <=256 batch×class groups.
- Parallel per-group score filter/bitonic sort plus bounded deterministic suppression; only 8-byte count D2H, then one device materialization. No full-input D2H or double selection.
- RTX 4060 GPU suite: 6 passed, 0 ignored; real CUDA plugin E2E: 1 passed; Unique policy regressions: 2 passed; exact CUDA/plugin Clippy passed.
- Representative B=1/C=2/N=32: 1 prepare, 1 count, 1 materialize launch; 16 rows, 272 B governed workspace, 8 B D2H.
- Seven requested mutations were caught. No throughput claim; capture and out-of-bound geometry fail closed.

## 2026-08-26 — #1896 synchronous H2D contract retained

Established that synchronous pageable H2D must complete device residency before return and order against the EP nonblocking compute stream; async overlap remains in `htod_async` with explicit fences. The initial revision was rejected because its schedule was not formal and Drop could panic; teardown paths must remain non-panicking.

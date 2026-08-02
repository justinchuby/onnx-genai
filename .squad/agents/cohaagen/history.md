# cohaagen — History

## 2026-07-29T00:45:00+0000 — PR #380 re-review
- Re-reviewed Melina's encoder-decoder fixture correction for issue #377 and approved PR #380, merged as `47c3331d`.
- Ran the CLI ORT E2E gate (23/23); metadata/I/O-detection reviews require that gate alongside engine/native unit tests.

## 2026-07-30T09:16:00Z — 7B native CUDA perf findings

- The Foundry baseline reports native CUDA ahead of ORT; 7B tracing localized o_proj at 19.5% of kernel time.
- Reverted the two-way o_proj split-K gate after a repeatable 0.59% 7B regression; do not retry that lever without a new higher-split kernel experiment.

## 2026-07-31T04:05:00Z — #87 first increment: async fence-ordered weight page-in (PR #544, draft)

- Switched live CUDA residency page-in (`resident_materialized`) from sync `cuMemcpyHtoD` to `upload_async` (pinned staging + `htod_async` on copy stream + `record_copy_fence`), consumer ordered via `compute_wait_fence`. `admit` now drains the compute stream ONLY when a page-in must evict (preserves WAR/reuse safety, allows overlap otherwise).
- Gate: `ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN` (default on; `=0` = old sync arm / kill-switch). Offload-off byte-identical.
- Anti-regression GPU test: `provider::tests::async_pagein_fence_orders_weight_page_in_consumer` (reads POISON without the fence).
- A/B (Qwen3-0.6B int4 native CUDA, H200): decode ms/token off=3.79; on-sync/on-async ≈ 5.93–6.00 at every budget (2 MiB → 480 MiB). Async ~1% faster than sync only at 2 MiB; tie elsewhere. Tokens byte-identical off ≡ sync ≡ async.
- HONEST: ~1.56x decode tax is budget-independent for a dense sequential sweep — eviction (every page-in once over budget) forces a compute drain that re-serializes the transfer; per-page `materialize`+pinned-copy dwarfs raw H2D. Overlap needs increment-2 double-buffer look-ahead (prefetch next slot while computing, no eviction sync on critical path; bounded staging ring).
## 2026-08-02T03:18:00Z — Inc-1b PR-3 shipped and merged (#589)

- Scoped and built the capture-fold of the decode-inline sibling Executor as bucket-A, flag-gated default-OFF, with zero shared #443/#543 capture surface changes.
- Evidence: 2.05x native Qwen3.6-27B decode speedup (143.8 -> 70.1 ms/tok), byte-exact vs CPU fp32 oracle with capture engaged, and all 4 Harry invariants tested.
- Outcome: PR #589 approved by Harry and merged by the coordinator.

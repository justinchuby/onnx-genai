# cohaagen — History

## 2026-07-29T00:45:00+0000 — PR #380 re-review
- Re-reviewed Melina's encoder-decoder fixture correction for issue #377 and approved PR #380, merged as `47c3331d`.
- Ran the CLI ORT E2E gate (23/23); metadata/I/O-detection reviews require that gate alongside engine/native unit tests.

## 2026-07-30T09:16:00Z — 7B native CUDA perf findings

- The Foundry baseline reports native CUDA ahead of ORT; 7B tracing localized o_proj at 19.5% of kernel time.
- Reverted the two-way o_proj split-K gate after a repeatable 0.59% 7B regression; do not retry that lever without a new higher-split kernel experiment.

# Bryant — History (compacted 2026-07-29)

**Role:** CPU/CUDA operator-coverage author and reviewer for the Rust runtime — CPU-EP op kernels, ONNX shape-inference rules, and backend-conformance artifacts. Preserve fail-closed claims for unsupported dtypes/formats, independent layout verification, and reviewer-lockout ownership transfers.

## Durable lessons
- Reviewer rejection triggers strict lockout and transfers revision ownership; record it. Precedents: the `Unique` CPU kernel (O(n²)/NaN/String defects) went to Pris/Deckard/Sapper and landed as `6a7755c`; CUDA GQA flash-prefill (`Sq != Sk` causal defect) went to Rachael; the batch-4 shape catalog PR #346 (domain/dtype defects) went to Rachael's `c20ec211`.
- Unsupported IQ formats and dtypes must stay fail-closed. llama.cpp grids/block layouts (imported `b15ca938`) require independent verification — Leon 🟢 hand-traced them.
- Backend-conformance artifacts must be refreshed after every op-coverage batch; a coverage claim without regenerated artifacts is incomplete. Zero integer divisors follow the runtime's existing zero convention.
- oneDNN CPU GEMM (feature/kernel/build glue + submodule) was removed in `453d280`; MLAS is the optional CPU GEMM feature. Registry count must stay intact across such removals.
- Shape inference reached 205 operators / 247 versioned entries at PR #346; new rules must satisfy ONNX version/shape/dtype contracts plus symbolic, divisibility, and overflow cases.

## Recent work (current wave, ~2026-07-28)
## 2026-07-28T04-08-08+0000 — Wave 2 regression/roadmap update
- Reviewed PR #311, requested changes, then re-approved Daniels' BF16 CUDA fix; #67 advanced to next batch.

## 2026-07-28T11:20:06+0000 — #75 batch 4 lockout outcome
- Authored PR #346 (`f53ed934`) adding the batch-4 catalog, advancing shape inference to 205 operators / 247 versioned entries.
- Holden requested changes for two domain/dtype correctness defects. Strict reviewer lockout transferred revision ownership to Rachael; her independent `c20ec211` correction was re-approved by Holden.

Full pre-compaction history in `history-archive.md`.
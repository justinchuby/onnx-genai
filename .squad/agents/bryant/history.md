# Bryant — History

## 2026-07-16T17:00:38+0000 — IQ super-block CPU support
- Merged `f6c530f`, adding native `BlockQuantizedMatMul` decoding for IQ4_XS, IQ3_S, and IQ2_XXS.
- Leon 🟢 verified the imported llama.cpp `b15ca938` grids and block layouts; unsupported IQ formats remain fail-closed.

## 2026-07-16T18:11:48+0000 — Complete CPU IQ family

- Merged `2dfee14` (IQ2_XS/IQ2_S/IQ3_XXS) and `1bf47a8` (IQ1_S/IQ1_M), completing runtime CPU decode for all supported IQ formats.
- Leon 🟢 independently verified llama.cpp layouts, grids, fingerprints, and hand traces.

## 2026-07-16T23:30:00+0000 — Pad axes shape-inference review

- 🟢 Cleared Joi's `0a105a4`: Pad applies opset-18 axes in order, preserves other dimensions, and normalizes negative axes.
- The expanded-Attention regression proves `[2,3,4,6]` / 576 bytes; the later Less Float32-vs-Bool inference fault is pre-existing.

## 2026-07-17T00:19:41+0000 — CPU ONNX Mod review

- 🟡 Advisory-cleared Joi's `aa7127e`: arithmetic and broadcasting are correct. Zero integer divisors follow this runtime's existing zero convention; add direct BF16 coverage.


## 2026-07-17T00:58:13Z — Chew logical and Expand reviews

- 🟢 Cleared `557ca87`: CPU Bool `And`/`Or`/`Xor`/`Not` use logical nonzero semantics, canonical outputs, and broadcast truth-table coverage; 436 CPU tests passed.
- 🟢 Cleared `14b5136`: `Expand` covers both broadcast directions, strict incompatibility, dtype passthrough, and unknown target-value rank fallback; 120 shape-inference tests passed.

## 2026-07-17T02:24:32Z — Standard shape-inference review

- 🟢 Cleared `98ee7a6`: the five new rules satisfy ONNX version/shape/dtype contracts and symbolic, divisibility, and overflow cases; 140 tests passed.

## 2026-07-17T07:19:39Z — onnx-rs multi-device/sharding review

- 🟢 Cleared Sapper's `be68145` multi-device/sharding proto integration and Deckard's `b5ccd3c` optional-dimension checker correction.
- The landing also includes the ONNX v1.20/IR13 spec-coverage audit; remaining gaps are tracked for Sapper.

- 2026-07-18 Scribe: Reshape/Split review and approval recorded; final fixes landed in 4ff24cb.

- 2026-07-19T12:40Z: Refreshed ONNX backend-test conformance artifacts; live CPU node coverage is 875/1,765 passing at ec5118c (890 failing; CUDA variants skipped).

## 2026-07-19T13:35Z — test-staleness guard
- 🟢 Approved Pris's unsupported-op sentinel guard after verifying `NxrtNeverRegisteredSentinelOp` remains unregistered and 23/23 executor tests pass; landed as `6ba4d96`.
## 2026-07-19T14:10Z — CPU scan/window op coverage
- Landed `5816d23`: CumSum opset-14 fix, CumProd opset-26, and Hann/Hamming/Blackman windows. Chew 🟡 approved with nits; backend node suite later reached 921 passing.


- **2026-07-19T16:15:00Z — CPU-EP coverage:** Added Selu, ThresholdedRelu, and LpNormalization; after dtype/f64 review fixes, refreshed backend conformance artifacts to 936/829/1765 in `4c05ede`.


## 2026-07-19T18:20:00Z — CPU-EP op coverage 936→975

- Added LpPool/GlobalLpPool/SpaceToDepth (`62fcb62`) and refreshed backend conformance artifacts to 975/790/1765 (`eef2c81`).


## 2026-07-19T20:10Z — CPU-EP op coverage Batch 4

- Added deterministic Dropout opsets 13/22 and Split opsets 1/18 (`4565e68`); Chew approved.
- Refreshed backend conformance artifacts to 1,012 passed / 753 failed / 1,765 skipped (`8c2a264`); Gaff approved.

- 2026-07-19: Authored initial Unique CPU kernel. Review found O(n²) grouping, NaN semantics, and String coverage issues; revision ownership transferred under lockout. Final numeric/bool/bf16 implementation landed as 6a7755c after Pris/Deckard/Sapper revisions and Luv approval.

## 2026-07-19T21:30:00Z — oneDNN backend removal
- Removed the oneDNN CPU GEMM feature/kernel/build glue and submodule in `453d280`; merged to `origin/main`.
- 620 CPU-EP library tests passed and registry count stayed intact; Luv approved the removal.

## 2026-07-20T05:20:00Z — CPU MatMul warning cleanup

- Removed the dead `out_start` binding and unused `mut` from `matmul.rs` (`984c239`), leaving default and `--features mlas` builds warning-clean; lightweight verification passed.


## 2026-07-20T13:35:00Z — Multistream performance and issue #40

- Authored the initial CUDA GQA flash-prefill integration; Chew rejected an `Sq != Sk` causal-origin defect and locked the artifact, after which Rachael owned and landed the correction.

## 2026-07-21T03:15:00Z — CUDA graph M4 validated
- Made MatMulNBits M=1 decode capture-safe (`a210703`) and closed GQA's final safety blocker with a sticky detect-before-consume error latch (`ca50bae`).

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.

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

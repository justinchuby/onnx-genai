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

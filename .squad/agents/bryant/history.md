# Bryant — History

## 2026-07-16T17:00:38+0000 — IQ super-block CPU support
- Merged `f6c530f`, adding native `BlockQuantizedMatMul` decoding for IQ4_XS, IQ3_S, and IQ2_XXS.
- Leon 🟢 verified the imported llama.cpp `b15ca938` grids and block layouts; unsupported IQ formats remain fail-closed.

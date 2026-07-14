# Session Log — nxrt rename + GELU fusion + crate reservation

**Timestamp:** 2026-07-14T14:50:00Z
**Agents spawned:** batty-19, roy-21, leon-16, gaff-16, deckard-23, roy-22, leon-17, roy-23
**Requested by:** Justin Chu (@justinchuby)

## Work Completed

### 1. GELU Erf-decomposition fusion (batty-19 / roy-21)
Fused bert_toy's GELU diamond `Mul(X,0.5)×(1+Erf(Div(X,√2)))` into `com.microsoft::Gelu` v1. New CPU kernel reusing `erf` helper. Strict constant guards (±1e-6) + diamond closure on `ValueId`. Fires 6× on bert_toy; parity within bounds; all tests green. Merged as `8e8d806`. **Phase-2 OpFusion set: LayerNorm / FusedMatMulBias / FusedGemm / FusedAttention / Gelu — all complete.**

### 2. Product rename ort2 → nxrt C-ABI symbols (leon-16 / gaff-16)
Renamed 17 `ort2_*` → `nxrt_*` `extern "C"` symbols in capi (lib.rs + tests/capi.rs). Zero `ort2_` remaining in `crates/`. Preserved `docs/ORT2.md` citations and `ort2-session`/`ort2-ep-api` label strings (intentional, out of scope). No alias shims. Merged as `43292ee`.

### 3. Crate-name reservation prep + cycle fix (deckard-23 / leon-17, two commits)
- `8988abd`: All 8 `onnx-runtime-*` crates → `0.1.0-dev.0`; exact `=0.1.0-dev.0` workspace pins; `docs/CRATE_RESERVATION.md` runbook.
- Roy (roy-22) 🔴 RED: shape-inference ↔ loader publish cycle in the documented order. Deckard locked out.
- `183a876`: Leon (leon-17) fixed by making shape-inference's dev-dep on loader path-only (no version). Roy (roy-23) 🟢 GREEN.
- Actual crates.io upload blocked pending user-provided token.

## Inbox Processed

- `deckard-crate-reserve.md` ✅
- `gaff-nxrt-rename-review.md` ✅
- `roy-crate-reserve-review.md` ✅
- `roy-crate-reserve-rereview.md` ✅

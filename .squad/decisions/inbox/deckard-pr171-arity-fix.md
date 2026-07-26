### 2026-07-26: Preserve aliased logical operands in SiluMul fusion
**By:** Deckard
**What:** Build fused `SiluMul` inputs from the matched `Silu` and `Mul` operand positions, preserving `[x, x]` for `Mul(Silu(x), x)` instead of using the deduplicated external-input list.
**Why:** Repeated graph values are valid distinct logical operands. Preserving both references keeps the common aliased gate fused, satisfies the two-input kernel contract, and retains byte-identical sequential SiLU-then-Mul semantics.

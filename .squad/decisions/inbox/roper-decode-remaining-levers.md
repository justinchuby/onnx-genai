# Decision: Decode's two remaining big-build levers — build B (capture-stable verify) first

**Author:** Roper (Architect) — landed by coordinator (read-only tools)
**Date:** 2026-08-14
**Slug:** roper-decode-remaining-levers
**Related:** #928, #932, #933, #935, spec-14b verdict; doc `docs/research/decode-remaining-levers-feasibility.md`

## Question
After the three cheap decode levers were closed (kernel micro-opt +2.7% Amdahl-capped; TP NO-GO for speed; eager-M=K speculative KILL), which of the two remaining multi-week "big build" levers should decode effort go to?

- **Lever A — Marlin int4 weight relayout:** improves the GEMV kernel (16B cp.async.cg + LOP3 dequant). Amdahl-capped over the ~61% GEMV fraction → **~1.3–1.6×, unconditional**. SM80+ only, two weight layouts to maintain.
- **Lever B — capture-stable padded M=K verify graph:** a fixed-shape padded verify sub-graph that REPLAYS instead of re-launching. Attacks the actual dispatch binding. Key insight: a captured M=K replay reads weights ONCE (weight-DRAM is per-token, not per-M), so it amortizes BOTH dispatch AND weight-read over K tokens → **floor ≈1.0×, ceiling ~2–3×**. Arch-agnostic core; reuses existing frozen-bucket capture machinery; unlocks prompt-lookup AND EAGLE-3/MTP from one build.

## Decision
**Build B first**, gated on a cheap Phase-0 capture-stability probe (prove a padded M=K graph instantiates capture-safe, replays ~1 dispatch/verify across bucket growth, and costs ≈ one M=1 replay). Keep A funded as the unconditional (~1.5×) fallback / parallel lever; A becomes primary only if B's Phase-0 fails.

Rationale: B attacks the dispatch binding (the settled diagnosis: decode is dispatch-bound, capture is load-bearing) while A only improves work the capture already amortizes; B's floor ≈1.0× means it is never a regression once capture-stable (unlike today's eager verify), and the spec-14b result proved that at 96% acceptance speculative would win big IF verify didn't break capture — B is exactly that fix.

## Go/no-go
- **B:** 🟢 GO to Phase-0; GO to build if Phase-0 passes. NO-GO if M=K kernels can't be made capture-safe OR replay cost scales ~K× not ≈1×.
- **A:** 🟡 CONDITIONAL GO. Full GO if an M=1 Marlin GEMV microbench lifts achieved DRAM 29% → ≥~55%. NO-GO if it stays <~40% (fold-scale territory).

## Next step
Run B's Phase-0 probe (throwaway `#[ignore]` capture-stability microbench) before committing eng-weeks. Details in the doc §3.

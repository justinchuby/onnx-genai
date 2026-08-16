# Session Log — PR #762 Third Corrective Wave

**Date:** 2026-08-11T21:00:00Z
**Branch:** `squad/ep-plugin-parity-cuda`
**PR:** #762

## Summary

Third rejection of #762 and its correction. Fourteen agents across fixing, reviewing, and hardening rounds.

**Root findings:**
1. EP was declining optional-slot nodes; BL2 fix was dead code in the ORT plugin path (Mariette).
2. `__absent_output_*` string sentinel was forgeable from model content (Challenger → Coco).
3. `filter_map` on shape dims destroyed rank at multiple sites (Challenger → Coco).
4. `Session_GetEpGraphAssignmentInfo` deferral was false — API present since ORT 1.24 (Fact Checker → Resch).
5. BL1 regression test lacked fallback guard (Pris → Rachael).

## Final state

- EP crates: 269 passed / 0 failed
- Workspace: 4598 passed / 20 failed / 436 ignored (20 pre-existing on base `675b697bc`)
- Five EP crates clippy-clean; fmt clean
- PR marked ready for review

## Agents

Leon, Sebastian, Isidore, Freysa, Luv, Mariette, Challenger, Coco, Fact Checker, Resch, Pris, Rachael, Zhora, Gaff

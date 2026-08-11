# Decision: Final documentation pass for PR #762

**Date:** 2026-08-11
**Author:** Roy
**Scope:** PR #762 (`squad/ep-plugin-parity-cuda`)

## Context

PR #762 was signed off by independent Opus review with no blockers. Before undrafting, documentation must accurately reflect the final state — particularly the CUDA EP status which shifted from "implementation-blocked" to "hardware-blocked" after the B1–B4 fixes landed.

## Changes Made

1. **NXRT_ABI.md** — Updated stale HEAD SHA `087d34888` → `fb9d757b3` in 4 locations.
2. **EP_PLUGIN_EXPORT_INVENTORY.md** — Changed CUDA status from 🔴 IMPLEMENTATION-BLOCKED to 🟡 HARDWARE-BLOCKED (3 locations). All four defects are resolved in code; what remains is GPU validation (#768).
3. **EP_PLUGIN_EXPORT_PR.md** — Updated stale HEAD SHA `087d34888` → `fb9d757b3` in 2 locations.
4. **PR body** — Complete rewrite via `gh pr edit 762 --body-file`. Describes CPU as proven (23 ORT tests), nxrt ABI as proven (10 round-trip tests), CUDA as structurally complete but unvalidated on hardware. No overclaims.

## Decision

CUDA EP documentation now says "hardware-blocked" (not "implementation-blocked") because the four code defects are fixed. The distinction matters: implementation-blocked implies the code is wrong; hardware-blocked means the code is untestable without a GPU.

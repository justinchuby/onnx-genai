# Session Log — Rejection-Response Wave

**Timestamp:** 2026-08-12T03:00:00Z
**Branch:** `squad/ep-plugin-parity-cuda`
**Requested by:** @justinchuby

## Outcomes

| Agent | PR | Task | Result |
|-------|----|------|--------|
| Leon | #32001 | B1 (arm64 detection) + N1-N4 | ✅ Fixed. Head `0d924a421b` |
| Mariette | #31988 | B1 admission, B2 occupancy, B3 tests | ✅ Fixed. Head `dc1e173e4b`. **Parked — GPU needed** |
| Coco | #31993 | NaN sNaN quieting, RNE tie, -march flag | ✅ Fixed. Head `02a9f34` |
| Deckard | #32003 | Strict-aliasing split (draft) | ✅ Opened draft. Incomplete fix found |
| Isidore | #32003 | Complete aliasing fix (4 `vec_a` sites) | ✅ Fixed. Head `23dcfddaaf` |
| Batty | #31988 | Build fix: sm_count parameter mismatch | ✅ Fixed. Commit `55e438ca6f` |
| Challenger | #32001 re-review | Found B-NEW-1 (false positive) | ⚠️ False positive disproved |
| Coordinator | #32001 | PR body rewrite (B2) + 5-case harness | ✅ Done |

## Milestone

**PR #31985 MERGED** (`f2dfa4e9eb`, 2026-08-12T00:49:43Z) — first upstream contribution landed.

## PRs parked

- **#31988**: Parked pending GPU access (CC 8.6/8.9 consumer + CC 8.0/9.0 datacenter needed).

## Key lessons

1. Separate admission from launch; pin acceptance set with exhaustive regression.
2. Reviewer blockers can be false positives — verify with same rigor as author claims.
3. Hardware quiets sNaN; assert with quiet-bit masking for NaN, raw-bit for non-NaN.
4. A flag fix can be incomplete — grep for the pattern, not just the first occurrence.
5. A corrected claim can itself be wrong (fp16 cast `-march` flag was unnecessary).

# Decision: PR #762 Documentation Accuracy Pass

**Author:** Zhora (Server Dev / API)
**Date:** 2026-08-11T19:30Z
**PR:** #762 (`squad/ep-plugin-parity-cuda`)
**Trigger:** @justinchuby requested documentation-only accuracy pass before undrafting

---

## Context

PR #762 went through six independent review rounds. The PR body and docs were written after round two and had become materially stale — they did not reflect the three late-breaking blockers fixed in rounds 3–6, the substantially stronger test story (EP assignment proof via `Session_GetEpGraphAssignmentInfo`), or the corrected test numbers (269 EP tests, not 142+23).

## Changes Made

### PR body (complete rewrite)

- Removed outdated M1/M2 split structure and "push is blocked" notice
- Documented all three blockers found since round 2 (optional slot fidelity, LayerNorm axis, forgeable sentinel)
- Documented the test story upgrade: `disable_cpu_ep_fallback`, `Session_GetEpGraphAssignmentInfo`, 14 assignment assertions, falsifiability proof
- Used exact verified numbers: 269 EP tests passing, 4580/20/436 workspace
- Explicit "What Is NOT Proven" section: CUDA is unvalidated on hardware, no mock = hardware evidence, #768 tracks it
- Noted the six review rounds honestly — what they found and what that means for confidence

### Docs updated

| File | Change |
|------|--------|
| `docs/NXRT_ABI.md` | 8 stale SHA refs (`fb9d757b3`, `62f23440f`) → `c1d2556b5` |
| `docs/EP_PLUGIN_EXPORT.md` | Stale commit ref `bad3682` → `c1d2556b5` |
| `docs/EP_PLUGIN_EXPORT_PR.md` | Header rewritten from "REJECTED" to accurate 6-round status; all `fb9d757b3`/`62f23440f` refs → `c1d2556b5` |
| `docs/EP_PLUGIN_EXPORT_INVENTORY.md` | Verification note SHA `62f23440f` → `c1d2556b5` |

### What was NOT changed

- `docs/CUDA_EP_STATUS.md` — already accurate. It correctly states all four defects are "Fixed (unvalidated on hardware)" and references #768. No stale claims found.
- `docs/EP_PLUGIN_EXPORT_ABI_TRUTH.md` — factual reference to ORT headers; no stale claims.
- `docs/EP_PLUGIN_EXPORT_SECURITY_AUDIT.md` — not swept (out of scope for this pass).
- No code, tests, or crate sources were modified.

## Constraints Respected

- PR remains draft
- No CUDA hardware claims made
- No mock presented as hardware evidence
- Honest about the six-round review process

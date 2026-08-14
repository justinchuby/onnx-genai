# Roy — History (compacted 2026-08-11T12-05-00Z)

**Role:** Architecture/planning and implementation reviewer spanning engine phases, ORT2 shape/optimizer work, EPContext, packaging, router design, CLI contracts, and stress-test design. Honor reviewer lockouts, keep documented contracts aligned with executable behavior, and preserve model/vendor/EP-agnostic interfaces.

## Durable lessons
- Engine work should keep moving away from monoliths toward explicit backend/sampler/proposer seams; runtime owns KV and public contracts must match executable behavior.
- EPContext generic encoding must stay model-agnostic and byte-exact; Roy's EP-literal encoder v1 was rejected and Deckard v2 is canonical.
- Pre-final docs written mid-work will always be stale. Always re-run validation commands at head and quote actual output — do not copy numbers from coordinator memo on faith.
- Implementation claims require quoted command output as evidence. "Passes locally" is not evidence; command transcript is.
- An agent's self-report is not evidence. Sapper reported all four CUDA defects fixed; review found use-after-free and unreachable success path.
- CUDA EP was "implementation-blocked" (not "hardware-blocked") until all four code defects were fixed. The distinction matters: implementation-blocked means the code is wrong; hardware-blocked means the code is untestable without a GPU.

## Historical context

Pre-2026-08-10 entries in `history-archive.md` (sections: July waves, ORT2 shape, EPContext, QMoE, CSA, CUDA M2, CPU serving, CLI).

2026-08-10/11 ep-plugin-export + early parity-cuda work archived in `history-archive.md` under "Archive batch 2026-08-10/11".

## Current entries (wave: ep-plugin-parity-cuda, 2026-08-11 CUDA B1-B4 fixes)

### 2026-08-11T06:34Z — PR #762 rejection response: doc rewrite for B1-B4 corrective wave

Corrected docs after rubber-duck review rejected PR #762 with four blockers. Reframed CUDA from "hardware-blocked" to "implementation-blocked."
- `docs/execution/CUDA_EP_STATUS.md`: Rewrote to CODE EXISTS/STUB/VALIDATED table. Four defects documented as specification. Fail-closed state recorded.
- `docs/ep-plugin/EP_PLUGIN_EXPORT_PR.md`: Recorded rejection with B1–B4 details. Validation updated to `62f23440f`: 231 tests, 1 ignored (LayerNorm Mean).
- `docs/architecture/NXRT_ABI.md`: Added inline-buffer rule and c_char portability rule. Test counts: 32/32 ABI + 10/10 host.
- `docs/ep-plugin/EP_PLUGIN_EXPORT_INVENTORY.md`: CUDA changed from 🟡 SCAFFOLDED to 🔴 IMPLEMENTATION-BLOCKED.
- Validated at `62f23440f`: 231 passing, 0 failures, 1 ignored.

### 2026-08-11 — Final documentation pass (PR #762 pre-undraft, commit `730889b94`)

Fixed stale SHAs and corrected CUDA status terminology after B1-B4 fixes landed.
- `docs/architecture/NXRT_ABI.md`: 4× SHA `087d34888` → `fb9d757b3`.
- `docs/ep-plugin/EP_PLUGIN_EXPORT_INVENTORY.md`: CUDA IMPLEMENTATION-BLOCKED → 🟡 HARDWARE-BLOCKED (3 locations).
- `docs/ep-plugin/EP_PLUGIN_EXPORT_PR.md`: 2× SHA updated.
- PR #762 body: complete rewrite — CPU proven (23 ORT tests), nxrt proven (10 roundtrip), CUDA unvalidated on hardware.

### 2026-08-11 — Session update (Scribe append)
PR #762 marked ready-for-review. 15 CI checks green. Upstream PRs #31973 and #31974 marked ready-for-review. `.squad/` git history purge complete on both upstream branches.

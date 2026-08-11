# Decision: ARM Upstream Inventory — Decline

**Date**: 2026-08-11
**Author**: Luba
**Status**: Recommendation (pending @justinchuby review)

## Decision

No ARM upstream candidate survives the combined filter of impact, novelty, testability (no ARM hardware), and acceptance likelihood. Upstream ORT's ARM support is already comprehensive (50+ MLAS files, NEON+SVE+KleidiAI). Our NEON work is Rust, not C++, so not directly upstreamable. Recommend declining ARM upstream track.

## Evidence

See `docs/UPSTREAM_ORT_ARM_INVENTORY.md` for full analysis with file citations.

## Impact

Frees ARM upstream slot. Energy may be better directed at QNN EP work or contributing to in-flight SVE PRs (#31143, #31146).

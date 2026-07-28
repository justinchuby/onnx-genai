# Bishop History

- 2026-07-18T05:55:00Z — Hardened CSA CPU factory/planner validation and coverage (`a4e2c6d`); Deckard rejected follow-up because CPU provider claim gate still needed `supports_op` validation.

- 2026-07-27T10:09:19Z: Roadmap wave landed: #239/#246/#249/#248/#256/#263/#259 plus fmt gate #264; reviewer-lockout protocol enforced where required.

## 2026-07-27T13:12:20+00:00 — Roadmap wave-5

- Reviewed PR #267: requested missing bf16 coverage, then approved Batty's ragged causal/non-causal bf16 CPU-oracle parity tests and the merge.

## 2026-07-27T16:44:54Z — Wave 8 update
- Approved Moss PR #273 for #79 CUDA BlockQuantizedMoE kernel; merged as 4e4dd25d.

## 2026-07-27T16:44:54Z — Wave 9 update
Requested changes on PR #283 and assigned Batty as fix owner; Dallas is locked out from revising.
- 2026-07-27T16:44:54Z — Re-reviewed PR #283 after Batty fix and APPROVED. Mutation-proven tests pin single unsuffixed `controlnet_cond`, no runtime `conditioning_scale`, and loud multi-ControlNet failure. PR merged as 687612f5; #50 closed.

## 2026-07-27T19:35:00Z — Roadmap wave update
- Reviewed PR #288, requested changes, then approved after Deckard hardened LogSoftmax/BitShift tests.

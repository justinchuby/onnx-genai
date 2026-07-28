# Team Focus — now

**Current focus:** CUDA/CPU parity, scheduler coverage, metadata consumption, and route-first MoE offload.

**MERGED THIS WAVE:** PR #373 (`61d3bdac`) shipped issue #231 declarative name-agnostic I/O detection, operator-agnostic shared-KV contracts, and strict attention sequence-length validation. PR #378 (`ac75e146`) fixed flaky QMoE offload residency tests under coverage; its honest scheduler bounds and poison-recovering metrics lock prevent phantom-regression CI noise on unrelated PRs.

**OPEN / PARTIAL:**
- #355 remains open for Sequence/Optional/Map container typing, which requires an IR type-model extension.
- #55 remains open for heterogeneous per-node planning, JSON/YAML adapters, and non-CUDA accelerators.
- #82 is done. #87 and #63 need GPU-side prefetch unification and remain blocked on Phase-3b live device binding.

**BLOCKED GAPS:**
- Real large-model E2E offload remains blocked by external GPU memory occupation, the 27B native Unsqueeze rank bug, and unmerged Mobius #432.
- Granite's unfused MatMul MoE export does not engage route-first QMoE offload.

**OFF-LIMITS (other team):** #54 model-package, #299 LoRA, and resting other-squad open PRs (#314, #315, #317, #318, #291, #99).

**Updated:** 2026-07-28T21:15:00+0000

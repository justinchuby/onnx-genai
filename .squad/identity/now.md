# Team Focus — now

**Current focus:** CUDA/CPU parity, scheduler coverage, metadata consumption, and route-first MoE offload.

**MERGED THIS WAVE:** PR #373 (`61d3bdac`) shipped issue #231 declarative name-agnostic I/O detection, operator-agnostic shared-KV contracts, and strict attention sequence-length validation. PR #378 (`ac75e146`) fixed flaky QMoE offload residency tests under coverage; its honest scheduler bounds and poison-recovering metrics lock prevent phantom-regression CI noise on unrelated PRs. PR #380 (`47c3331d`) landed #377's CORE name-agnostic model pipeline: decode roles use explicit metadata or a unique shape match, and ambiguous encoder-decoder fixtures declare component I/O.

**OPEN / PARTIAL:**
- #355 remains open for Sequence/Optional/Map container typing, which requires an IR type-model extension.
- #55 remains open for heterogeneous per-node planning, JSON/YAML adapters, and non-CUDA accelerators.
- #82 is done. #87 and #63 need GPU-side prefetch unification and remain blocked on Phase-3b live device binding.
- #377 remains open for deferred name-guess adapters: ORT shared/static-cache, MTP/EAGLE3/Gemma4/shared-KV proposers, speculative target discovery, the paged-KV bridge, and nested autoregressive routing.

**BLOCKED GAPS:**
- Real large-model E2E offload remains blocked by external GPU memory occupation, the 27B native Unsqueeze rank bug, and unmerged Mobius #432.
- Granite's unfused MatMul MoE export does not engage route-first QMoE offload.

**OFF-LIMITS:** #54 model-package and #299 LoRA belong to another team; #106 is under Justin's study. Do not touch resting other-squad open PRs (#314, #315, #317, #318, #291, #99).

**Updated:** 2026-07-29T00:45:00+0000

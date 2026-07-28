# Team Focus — now

**Current focus:** CUDA/CPU parity, scheduler coverage, metadata consumption, and
route-first MoE offload.

**MERGED THIS WAVE:** PR #362 (`5a079029`) advances #355 with If/Loop/Scan
shape inference; PR #364 (`3b08c025`) establishes guarded route-first QMoE
host prefetch; PR #365 (`83f4c293`) consumes `onnx_runtime.*` metadata hints
with structural node identity.

**OPEN / PARTIAL:**
- #355 remains open for Sequence/Optional/Map container typing, which requires an
  IR type-model extension.
- #55 remains open for heterogeneous per-node planning, JSON/YAML adapters, and
  non-CUDA accelerators.
- #82 is done. #87 and #63 need GPU-side prefetch unification and remain blocked
  on Phase-3b live device binding.

**BLOCKED GAPS:**
- Real large-model E2E offload: the 27B native path has an Unsqueeze rank bug;
  ORT misses `past_key_values.*.recurrent_state`; Mobius #432 is unmerged; and
  external VLLM work occupies GPU memory.
- Granite's unfused MatMul MoE export does not engage route-first QMoE offload.

**OFF-LIMITS (other team):** #54 model-package, #299 LoRA, and resting
other-squad open PRs (#314, #315, #317, #318, #291, #99).

**Updated:** 2026-07-28T17:40:00+0000

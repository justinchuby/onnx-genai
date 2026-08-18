# Decision: placement viability on the native CUDA path (item 3 of the offload directive)

- **Date:** 2026-08-18
- **Author:** copilot (CLI)
- **Scope:** native CUDA path only (NOT plugin-EP; #982 untouched)

## Context
Standing directive: advance vmm/offload/streaming/batching for large models. Took `offload`. The one open, unproven lever in `MEMORY_MANAGEMENT_MODEL_DESIGN.md` was **placement** (running an op where its weights live when arithmetic intensity is low), gated by an "unknown": is a per-token device excursion compatible with CUDA graph capture on the native path (#994/#854/#867)?

## Decisions / findings
1. **Capture compatibility = YES, as an eager seam.** A per-token device→host→device excursion is compatible with native CUDA graph capture *only* between captured segments (the existing segmented-capture machinery), and is illegal inside an active capture (host-consuming D2H needs a sync → invalidates capture). Locked in by permanent tests in `crates/onnx-runtime-ep-cuda/src/graph.rs` (PR #1297).
2. **Seam price ≈ 47–90 µs/token** on RTX 4060 (bit-identity asserted). Negligible vs a bandwidth-bound weight stream.
3. **The doc's placement target does not exist on `qwen14b-zp`.** The streamed 389 MB/token (paging key 919) is `lm_head.weight` (MatMulNBits vocab projection), verified by lazy-handle dump. The embedding gather (`model.embed_tokens.qweight`, `GatherBlockQuantized`) is **not a lazy boundary → resident → streams nothing**. Host-placing it saves nothing. Build was **not** started. See issue #1299.
4. `lm_head.weight` is itself memory-bound and *might* be a compute-on-host candidate, but that is a different op (needs a host INT4 GEMV) with its own go/no-go — deferred to the owner.

## Follow-ups shipped alongside
- #1266 clamp follow-up: option C (document ceiling + issue #1288), PR #1289 merged.
- 3 known `model.io.token_input` smoke failures: harness root cause, issue #1284.
- Capture-seam correctness + seam-price bench: PR #1297 merged.

## Note
4 pre-existing RTX 4060 kernel bit-exactness failures (`matmul_nbits` SwiGLU fused vs reference off-by-1 in fp16; one `group_query_attention` gate assertion) reproduce in isolation and are unrelated to this work.

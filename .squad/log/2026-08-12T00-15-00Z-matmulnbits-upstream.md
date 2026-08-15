# Session Log — MatMulNBits Upstream Workstream

**Timestamp:** 2026-08-12T00:15:00Z
**Branch:** nxrt/cuda-matmulnbits-sm-cols (upstream ORT)
**PR:** microsoft/onnxruntime#31988 (draft)

## Summary

- **Provenance audit:** "+2.08% on 3 GPUs" claim had no benchmark record in-repo. Kept out of PR entirely.
- **Upstream gap confirmed:** `kColsPerThreadBlock = 8` hardcoded in M=1 GEMV with no SM-count adaptation. No colliding in-flight work.
- **Implementation:** `SelectColsPerBlock(n, sm_count)` → 8/4/2. M=1 kernel templated on `cols_per_block`. `multiProcessorCount` threaded through 3 layers. Bit-identical.
- **Routing guard (Chew):** Sebastian cleared n%8≠0 as "SAFE" — wrong. `n=12` with cols=4 would have been newly accepted. Guard added; accepted-shape set pinned with exhaustive test.
- **Fresh review (Gaff):** No blockers. Bit-identicality and routing invariance genuine.
- **Clang-format fix (Coordinator):** `186b89604c`. Cosmetic only.
- **CPU AMX: no PR.** Host is AMD EPYC 9V74 — no AMX, no VNNI — cannot run the required comparison.
- **Split-K: excluded.** 2-way K_SPLIT=2 regressed 7B o_proj GEMV by −0.59%.

## Reviewer lockout

Cohaagen (author) and Sebastian (reviewer) barred from routing-guard revision. Chew revised; Gaff reviewed fresh.

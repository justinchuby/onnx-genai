### 2026-08-11: Share exact cuBLASLt GEMM workspace as one persistent peak
**By:** Copilot
**What:** MatMul, Gemm, MatMulNBits, FusedMatMulBias, and FusedGemm report the selected cuBLASLt heuristic `workspaceSize` and share one session-persistent executor peak. Attention Phase-2a remains in its step-scoped composite buffer, with its direct-execute compatibility fallback unchanged.
**Why:** The 32 MiB constant is only the heuristic ceiling. RTX 4060 regression shapes selected 0-96 bytes, so fixed per-site reservations would substantially over-charge the device authority. Planning and execution use the same plan helper and reject any shortfall deterministically.

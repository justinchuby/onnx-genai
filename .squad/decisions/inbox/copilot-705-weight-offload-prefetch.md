### 2026-08-06: Weight offload defaults to async mmap-backed page-in
**By:** Copilot
**What:** CUDA weight offload now defaults to async, fence-ordered page-in and copies directly from external mmap regions into reusable pinned staging instead of materializing an owned host tensor per miss. The synchronous demand-copy path remains available with ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN=0 for A/B.
**Why:** On qwen2.5-14b int4 under WDDM, H2D itself was not the dominant cost; redundant host materialization and disabled prefetch kept the managed path far behind WDDM spill. The new default makes prefetch reachable in the not-fit regime and reports materialize/H2D/wait/sync counters so future measurements can verify the active arm.

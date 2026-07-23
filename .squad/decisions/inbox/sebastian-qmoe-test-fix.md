### 2026-07-23: Serialize QMoE GPU capture tests and verify live replay routing
**By:** Sebastian
**What:** QMoE integration tests now hold a process-wide GPU mutex for each test body. The capture test also changes `router_probs` after capture and compares replay against an uncaptured eager run using the new expert routes.
**Why:** Concurrent CUDA allocation can invalidate thread-local graph capture, while changed-routing parity proves expert selection is recomputed from live replay inputs rather than baked into the graph.

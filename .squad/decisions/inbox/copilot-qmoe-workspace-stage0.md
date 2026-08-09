### 2026-08-09: QMoE workspace is prepared before admission
**By:** Copilot
**What:** Kernels declare typed workspace requirements from concrete tensor metadata. Native CUDA prefill resolves QMoE shapes and reserves one reusable session-persistent workspace peak before the admission callback; execution receives only that prepared buffer.
**Why:** QMoE prompt-sized scratch previously allocated inside execute, after HTTP/SSE admission. Sharing the checked layout helper between planning and execution prevents formula drift and turns omissions or shape changes into explicit invariant failures.

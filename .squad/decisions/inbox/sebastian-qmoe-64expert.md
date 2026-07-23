### 2026-07-23: Add 64-expert top-6 CUDA QMoE parity coverage
**By:** Sebastian
**What:** Added parameterized synthetic 64-expert/top-6 QMoE GPU parity tests for fp16 decode (M=1) and prefill (M=8), bf16 decode/prefill, hot-expert plus empty-expert routing, capture warm/replay with changed routes, and a 64-row worst-case route-scratch allocation. Each uses the existing CPU QMoE oracle, except replay additionally compares against an uncaptured CUDA reference.
**Why:** DeepSeek-V2-Lite routing requires 64 experts and top-6, while the previous GPU tests only exercised 4 experts/top-2. GPU 5 results: qmoe_gpu 27 passed/0 failed; CUDA lib gate 192 passed/0 failed; clippy passed. No 64/top-6 kernel scale bug was found.

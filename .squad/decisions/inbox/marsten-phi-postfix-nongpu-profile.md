### 2026-07-23: Target the remaining Phi LongRoPE host If
**By:** Marsten
**What:** On fixed main with `719d2fe`, Phi has two captured graph regions
(`cuStreamBeginCapture=4` across two 128-token generations; 508 graph
launches = two per 254 decode forwards), 236.0 GPU kernels/decode-forward,
and zero graph fallbacks. Nsight reports 2.948 ms GPU kernels/token versus
5.150 ms/token uninstrumented wall time. The native op trace attributes a
1.935 ms median to the still-eager LongRoPE `If`; replayed `Greater` is only
1.28 us GPU/token and GQA is captured (0.406 ms GPU/token).
**Why:** Fully moving the branch select on-device is the highest-value
non-GEMV follow-up: its ~1.94 ms/token budget is about 88% of the ~2.20 ms
non-GPU remainder, with a 5.15 to ~3.2 ms/token theoretical ceiling. Kernel
launch batching is not first: the 236 kernels already arrive in two graph
launches per decode forward.

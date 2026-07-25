# Gorman review: post-#148 native-vs-ORT scorecard

**Verdict: 🟡 — mergeable after the review documentation update in this commit.**

All displayed arithmetic recomputes correctly: native/ORT is 1.385930
(1.386×), 1.605277 (1.605×), and 1.122583 (1.123×); the native leads are
38.593%, 60.528%, and 12.258%. The A/B changes are -0.08695%, +10.52325%,
and +2.07008%, respectively.

The ORT evidence is sufficient: CUDAExecutionProvider, 86–91% peak GPU
utilization, multi-GiB allocations, and no inserted-Memcpy warning are
recorded, with the intentional shape-node CPU notice distinguished from
partial-EP fallback. The A/B now explicitly records fixed prompt, tokens,
warmups, runs, steady window, CPU pinning, and GPU, with only #148 differing.

The larger Qwen1.5B gain is physically plausible: its smaller down-projection
K/N yields fewer baseline 8-column CTAs, hence greater H200 grid starvation;
the grid multiplier has more latency hiding to recover than for 7B. This is a
native-only A/B, not ORT jitter. GPU re-confirmation was not run: GPU 1 is
reserved (0% utilization but 129589 MiB allocated) and every other GPU was
98–99% utilized. The scorecard now states that a clean-idle-GPU Qwen1.5B
confirmation remains pending.

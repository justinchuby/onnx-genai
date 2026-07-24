### 2026-07-23: Record cumulative Phi prefetch and standalone int8 split-K frontier
**By:** Marsten
**What:** At `4e774ee`, Phi-4-mini reaches 193.32 tok/s (median of 7, 121.21--194.67 spread under shared-host contention), 15.81% behind the canonical ORT 0.14.1 reference, with zero fallbacks and coherent output. Qwen2.5-1.5B and DeepSeek-R1-Distill-Qwen-1.5B remain within noise at 617.90 and 622.66 tok/s.
**Why:** This is the honest cumulative frontier after stacking fused gate-up int4 software-prefetch and standalone int8-zp split-K; the median, full spread, and contention caveat prevent host variance from being misclassified as a regression.

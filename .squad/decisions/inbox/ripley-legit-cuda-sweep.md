### 2026-07-25: Require observed GPU execution for native-vs-ORT headlines
**By:** Ripley
**What:** Treat Phi-4-mini, Qwen2.5-1.5B, and Qwen2.5-7B as valid native-CUDA
versus ORT-CUDA comparisons for their Foundry `cuda-gpu` artifacts. Their ORT
runs had no inserted-Memcpy warning and were independently observed at 86–91%
H200 utilization. Report the real native wins as 1.385×, 1.452×, and 1.100×.
**Why:** Selecting the CUDA EP is insufficient proof by itself. A valid
competitive claim requires a CUDA-targeted artifact, absence of fallback-copy
thrash, and direct evidence that model compute exercised the selected GPU.

### 2026-08-13: Keep LRU default after MRU managed-path sweep
**By:** Copilot
**What:** MRU reduced H2D bytes/token in four pressured comparisons across Qwen2.5 14B and Qwen2 0.5B, but by a budget-sensitive 3.1% to 34.1%; keep the shipped LRU default and retain MRU as a probe.
**Why:** MRU is incremental to, and causally overlaps with, scan-resistant admission. It cannot affect bypassed tensors, which fail admission before victim selection and dominate the remaining recoverable gap. A naturally over-budget second large architecture and Linux reproduction are required before changing the default.

# Sebastian sampling top-k/top-p perf

2026-07-30 — Qwen3 sampling decode overhead was traced to per-token full-vocab sorts in `TopKProcessor` and `TopPProcessor`. The engine sampling path should avoid full-vocab `O(n log n)` sorting: top-k selects the threshold with partial selection, and top-p ranks only positive-probability survivors (using partition selection for large candidate sets) while preserving nucleus semantics. Perf claims for this change must report processor microbenchmarks and model-level greedy-vs-sampling decode throughput separately.

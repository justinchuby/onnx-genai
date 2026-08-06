### 2026-08-05: Dense prefetch stays eviction-neutral
**By:** Copilot
**What:** Executor-driven lazy-weight prefetch is scoped to dense MatMulNBits weights and CUDA admits a prefetch only when it fits without eviction or lease growth.
**Why:** The schedule should overlap dense transfers without changing MoE behavior or cache-victim selection; non-neutral budgets fall back to demand paging.
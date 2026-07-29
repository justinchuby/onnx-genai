### 2026-07-29: Registry-backed model warmup
**By:** Lull
**What:** Added an opt-in `warmup` per-model setting and `POST /v1/admin/models/{id}/warm`. Both use `ModelRegistry::warmup`, which performs one deterministic generated token and records a successful warmup idempotently.
**Why:** The first generation initializes lazy runtime allocations; sharing the registry method keeps configured and on-demand warmups identical while allowing failures to be retried without corrupting registry state.

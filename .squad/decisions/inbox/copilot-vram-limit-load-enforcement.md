### 2026-08-06: Enforce explicit VRAM limits at engine load
**By:** Copilot
**What:** Explicit byte VRAM limits are enforced in the engine load path. Native CUDA derives a weight-offload residency budget from the limit before constructing the CUDA EP; non-offload-capable backends fail at load when weights exceed the limit.
**Why:** The scheduler governor only derives KV budget, and the CUDA allocator sees allocations too late on WDDM. Engine load is the first point that knows the model package size, selected backend, offload capability, and the engine ledger that must remain authoritative.
**Update:** Derived CUDA offload budget is the VRAM limit minus device KV/recurrent state, and must meet the largest lazy-weight node working set. This prevents the auto-offload path from consuming the whole limit and failing later as a KV error.

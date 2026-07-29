### 2026-07-29: Preserve warmup error categories at the admin boundary
**By:** Rachael
**What:** `ModelRegistry::warmup` now returns typed absent-model, registry, and runtime-failure errors; the admin warm endpoint maps them to 404, 500, and 500 respectively.
**Why:** A loaded model's failed warmup must not be reported as an unloaded-model 404.

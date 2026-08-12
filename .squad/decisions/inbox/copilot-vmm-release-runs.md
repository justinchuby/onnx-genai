### 2026-08-12: Count VMM releases as driver unmap runs
**By:** Copilot
**What:** `GlobalVmmStats::releases` counts contiguous `cuMemUnmap` operations rather than individual granules released.
**Why:** Adjacent weight-page granules are now unmapped in one driver call. Keeping the old per-granule counter would hide whether release-side driver churn was actually reduced; committed-byte gauges continue to track the full released quantity.

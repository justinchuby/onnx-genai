### 2026-08-08: VMM weight admission uses mapped and owned constraints
**By:** Copilot
**What:** CUDA weight residency uses two authority-coordinated constraints: mapped granules consume the cache's weight allowance, while newly created handles consume global physical headroom. Content bytes remain separate cache-efficiency metrics; non-VMM residency retains content-byte admission.
**Why:** Mapping a retained pooled handle costs no new global ownership but still occupies weight-zone capacity. Already mapped shared granules cost neither. Admission must reserve both dimensions transactionally, and failed transactions must release newly created handles instead of retaining them in the pool.

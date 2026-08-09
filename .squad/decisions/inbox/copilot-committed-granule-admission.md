### 2026-08-08: VMM weight admission counts authority-owned granules
**By:** Copilot
**What:** CUDA weight residency uses transactional VMM span commitment and shared-pool physical ownership for admission. Content bytes remain separate cache-efficiency metrics; non-VMM residency retains content-byte admission.
**Why:** Mapping a retained pool handle consumes no new device ownership, while creating a handle consumes one granule. Admission must distinguish those operations and serialize the check with commitment to avoid late granule refusal.

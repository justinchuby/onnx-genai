### 2026-08-07: Physical backing uses one explicit accounting authority
**By:** Copilot
**What:** `MemoryGovernor` exposes a stable `MemoryAuthorityId`; backing-owned physical accounting names that authority, and `VirtualBuffer` rejects a different governor before reserving or committing memory.
**Why:** Pooled-unmapped physical bytes remain owned by the device ledger, while mapped holder/zone bytes are attribution only. Allowing different backing and buffer ledgers would bypass admission or double-charge the same physical memory.

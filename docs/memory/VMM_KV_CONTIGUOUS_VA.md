# Contiguous-VA KV investigation (historical note)

This file records the investigation path that established that live-length
bounded reads can leave a contiguous reservation tail uncommitted. Its
conclusions were subsequently extended and corrected by the seq-major and
token-major measurements.

The authoritative, current explanation of KV layout, VMM mapping geometry,
residency floors, crossovers, implementation status, and measured costs is:

**[`MEMORY_ARCHITECTURE.md` — KV layout and residency](MEMORY_ARCHITECTURE.md#kv-layout-and-residency)**

Keep future conclusions and numbers in that section rather than copying them
here. Historical implementation and test details remain available in PR #772.

# Seq-major KV investigation (historical note)

This file was the working investigation record for seq-major BSNH KV layout.
The design line later expanded through the landed decode pair and the
token-major probe, so retaining a second copy of the conclusions here would
allow the two accounts to diverge.

The authoritative, current explanation of KV layouts, residency, crossovers,
measurements, implementation status, prefix sharing, and per-EP ownership is:

**[`MEMORY_ARCHITECTURE.md` — KV layout and residency](MEMORY_ARCHITECTURE.md#kv-layout-and-residency)**

Keep future conclusions and numbers in that section rather than copying them
here. Historical benchmark and prototype details remain available in PR #778;
the narrow seq-major decode implementation landed in PR #782.
